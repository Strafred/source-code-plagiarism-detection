import java.util.Scanner;
public class Kasus3 {
    public static double hitungTinggi(double feet, double inches) {
        return feet * 12 + inches;
    }
    public static double hitungBmi(double weight, double feet, double inches) {
        return weight * 0.45359237 / ((hitungTinggi(feet, inches) * 0.0254) * (hitungTinggi(feet, inches) * 0.0254));
    }
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter weight in pounds: ");
        double weight = input.nextDouble();
        System.out.print("Enter feet: ");
        double feet = input.nextDouble();
        System.out.print("Enter inches: ");
        double inches = input.nextDouble();
        double bmi = hitungBmi(weight, feet, inches);
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
}
