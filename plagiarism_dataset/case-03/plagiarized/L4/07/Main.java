import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        double berat,feet,inches,height,bmi;
        System.out.print("Enter weight in pounds: ");
        berat = input.nextDouble();
        System.out.print("Enter feet: ");
        feet = input.nextDouble();
        System.out.print("Enter inches: ");
        inches = input.nextDouble();
        height = feet * 12 + inches;
        bmi = hitungBmi(berat, height);
        System.out.println("BMI is " + bmi);
        if (bmi < 18.5) {
            System.out.println("Underweight");
        } else if (bmi < 25) {
            System.out.println("Normal");
        } else if (bmi < 30) {
            System.out.println("Overweight");
        } else {
            System.out.println("Obese");
        }
    }
    public static double hitungBmi(double berat, double height){
        double bmi;
        bmi = berat * 0.45359237
                / ((height * 0.0254) * (height * 0.0254));
        return bmi;
    }
}
