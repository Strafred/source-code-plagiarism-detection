import java.util.Scanner;
public class Kasus3L3 {
    public static void main(String[] args) {
        System.out.print("Enter weight in pounds: ");
        Scanner inp = new Scanner(System.in);
        double berat = inp.nextDouble();
        System.out.print("Enter feet: ");
        double kaki = inp.nextDouble();
        System.out.print("Enter inches: ");
        double inch = inp.nextDouble();
        double tinggi = kaki * 12 + inch;
        double rata = berat * 0.45359237 /((tinggi * 0.0254) * (tinggi * 0.0254));
        System.out.println("BMI is " + rata);
        if (rata < 18.5)
            System.out.println("Underweight");
        else if (rata < 25)
            System.out.println("Normal");
        else if (rata < 30)
            System.out.println("Overweight");
        else
            System.out.println("Obese");
    }
}
