import java.util.Scanner;
public class Kasus3L5 {
    public static void main(String[] args) {
        System.out.print("Enter weight in pounds: ");
        Scanner inp = new Scanner(System.in);
        double berat = inp.nextDouble();
        System.out.print("Enter feet: ");
        double kaki = inp.nextDouble();
        System.out.print("Enter inches: ");
        double inch = inp.nextDouble();
        BMIKategori(hitungBMI(berat, kaki * 12 + inch));
    }
    private static double hitungBMI(double weight, double height){
        return weight * 0.45359237 /((height * 0.0254) * (height * 0.0254));
    }
    private static void BMIKategori(double BMI){
        System.out.println("BMI is " + BMI);
        if (BMI < 18.5)
            System.out.println("Underweight");
        else if (BMI < 25)
            System.out.println("Normal");
        else if (BMI < 30)
            System.out.println("Overweight");
        else
            System.out.println("Obese");
    }
}
